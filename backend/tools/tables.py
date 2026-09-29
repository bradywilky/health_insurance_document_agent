"""Deterministic table tools used by the document agent; no agent imports."""
import logging
from backend.shared.serialization import sanitize_result
import pandas as pd
from backend.storage.tables import load_table
from backend.tools.table_operations import query_table, join_tables

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _source_rows(frame, meta, indices):
    if meta.get('sheet_type') == 'metadata' and 'source_row' in frame:
        return [int(frame.iloc[i]['source_row']) for i in indices if i < len(frame)]
    rows = meta.get('source_rows', [])
    return [rows[i] for i in indices if i < len(rows)]

def execute_table_tool(
    tool_name: str,
    tool_input: dict,
    s3_client,
    s3_bucket: str,
    s3_prefix: str,
    plan_domain: str,
    filename: str,
    metadata: dict,
    call_log=None,
) -> dict:
    sheet_name = tool_input.get("sheet_name")
    sheets = metadata.get("sheets", {})

    if tool_name in {"read_sheet", "query_table"} and sheet_name not in sheets:
        return {"error": "Provide an exact sheet_name from list_sheets."}

    if tool_name == "query_table":
        df = load_table(s3_client, s3_bucket, s3_prefix, plan_domain, filename,
                      sheet_name, sheet_meta=sheets[sheet_name])
        result = query_table(df, {k:v for k,v in tool_input.items() if k != 'sheet_name'})
        indices = result.pop('contributing_row_indices') if result['aggregated'] else result['result_row_indices']
        result.pop('result_row_indices', None)
        result.pop('contributing_row_indices', None)
        result['sources'] = [{'sheet': sheet_name,
                              'source_rows': _source_rows(df, sheets[sheet_name], indices),
                              'csv_record_indices': indices,
                              'row_number_basis': 'CSV indices are zero-based; source rows are one-based'}]
        result['warnings'] = sheets[sheet_name].get('warnings', [])

    elif tool_name == "join_tables":
        left_name, right_name = tool_input.get('left_sheet'), tool_input.get('right_sheet')
        if left_name not in sheets or right_name not in sheets:
            return {'error': 'left_sheet and right_sheet must be exact names from the index.'}
        def load(name):
            return load_table(s3_client, s3_bucket, s3_prefix, plan_domain, filename,
                            name, sheet_meta=sheets[name])
        left_frame, right_frame = load(left_name), load(right_name)
        result = join_tables(left_frame, right_frame,
                             {k:v for k,v in tool_input.items() if k not in {'left_sheet','right_sheet'}})
        refs = result.pop('join_provenance', [])
        result['sources'] = []
        for name, key, frame in [(left_name,'__left_index',left_frame), (right_name,'__right_index',right_frame)]:
            indices = sorted({int(r[key]) for r in refs if pd.notna(r[key])})
            result['sources'].append({'sheet':name, 'source_rows':_source_rows(frame,sheets[name],indices),
                                      'csv_record_indices':indices})
        result.pop('contributing_row_indices', None)
        result.pop('result_row_indices', None)

    elif tool_name == "list_sheets":
        result = {
            name: {
                "sheet_type": s["sheet_type"],
                "row_count": s["row_count"],
                "column_count": s["column_count"],
                "columns": [
                    c["name"]
                    for c in s.get("columns", [])
                    if not c.get("likely_junk")
                ],
            }
            for name, s in sheets.items()
        }

    elif tool_name == "get_sheet_schema":
        sheet_meta = sheets.get(sheet_name)
        if sheet_meta is None:
            result = {
                "error": (
                    f"Sheet '{sheet_name}' not found. "
                    "Call list_sheets to see available sheets."
                )
            }
        else:
            result = {
                "sheet_name": sheet_name,
                "row_count": sheet_meta["row_count"],
                "columns": sheet_meta["columns"],
                "header_row": sheet_meta.get("header_row"),
                "context_rows": sheet_meta.get("context_rows", []),
                "warnings": sheet_meta.get("warnings", []),
            }

    elif tool_name == "read_sheet":
        offset = max(0, int(tool_input.get("offset", 0)))
        limit = max(1, min(100, int(tool_input.get("limit", 50))))
        smeta = sheets[sheet_name]
        df = load_table(s3_client, s3_bucket, s3_prefix, plan_domain, filename,
                      sheet_name, sheet_meta=smeta)
        end = min(offset + limit, len(df))
        result = {"sheet_name": sheet_name, "offset": offset, "total_rows": len(df),
                  "rows": df.iloc[offset:end].to_dict(orient="records"),
                  "next_offset": end if end < len(df) else None,
                  "source_rows": smeta.get("source_rows", [])[offset:end]
                      if smeta.get("sheet_type") == "data" else [],
                  "context_rows": smeta.get("context_rows", []),
                  "warnings": smeta.get("warnings", [])}

    elif tool_name == "search_all_sheets":
        term = tool_input.get("term", "")
        if not isinstance(term, str) or not term.strip():
            return {"error": "Provide a nonempty literal search term."}
        search_columns = tool_input.get("search_columns") or []
        results = {}
        details = {}

        for name, smeta in sheets.items():
            col_names = [
                c["name"]
                for c in smeta.get("columns", [])
                if not c.get("likely_junk")
            ]
            cols_to_search = (
                [c for c in col_names if c in search_columns]
                if search_columns
                else col_names
            )
            if not cols_to_search:
                continue

            try:
                df = load_table(
                    s3_client, s3_bucket, s3_prefix,
                    plan_domain, filename, name, sheet_meta=smeta
                )
                mask = pd.Series(False, index=df.index)

                for col in cols_to_search:
                    if col in df.columns:
                        mask |= df[col].astype("string").str.contains(
                            str(term), case=False, na=False, regex=False
                        )

                matched = df[mask][
                    [c for c in col_names if c in df.columns]
                ]
                if not matched.empty:
                    results[name] = matched.head(50).to_dict(orient="records")
                    source = smeta.get('source_rows', [])
                    details[name] = {'matches': len(matched), 'returned': min(len(matched), 50),
                                     'truncated': len(matched) > 50,
                                     'source_rows': [source[i] for i in matched.head(50).index if i < len(source)]
                                         if smeta.get('sheet_type') == 'data' else []}
            except Exception as e:
                logger.warning(
                    f"search_all_sheets: failed to search sheet '{name}': {e}"
                )
                details[name] = {'error': str(e)}

        result = (
            results
            if results
            else {"message": f"No matches found for '{term}' in any sheet."}
        )
        if details:
            result['_search_details'] = details

    else:
        result = {"error": f"Unknown table tool: {tool_name}"}

    return sanitize_result(result) if isinstance(result, dict) else {"result": result}
