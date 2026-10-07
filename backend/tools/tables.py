"""Deterministic table tools used by the document agent; no agent imports."""
import logging
from backend.shared.serialization import sanitize_result
from backend.shared.rows import narrative_rows
from backend.shared.table import Table, concat
from backend.storage.tables import load_table
from backend.tools.table_operations import query_table, join_tables

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _source_rows(frame, meta, indices):
    if meta.get('sheet_type') == 'metadata' and 'source_row' in frame.columns:
        return [int(frame.rows[i]['source_row']) for i in indices if i < len(frame)]
    rows = meta.get('source_rows', [])
    return [rows[i] for i in indices if i < len(rows)]

def _load_for_query(load, name, meta):
    """Narrative tabs are queried as one text line per sheet row, matching read_sheet and search."""
    frame = load(name)
    if meta.get('sheet_type') == 'metadata' and {'source_row', 'source_column', 'text'} <= set(frame.columns):
        frame = Table(['source_row', 'text'], [{'source_row': n, 'text': t} for n, t in narrative_rows(frame.rows)],
                      numeric=['source_row'])
    return frame


def _referenced_columns(params):
    names = set(params.get('select', [])) | set(params.get('group_by', []))
    for key in ('filters', 'aggregations', 'recode', 'sort', 'derive'):
        names |= {item.get('column') for item in params.get(key, []) if isinstance(item, dict)}
    derived = {item.get('as') for item in params.get('derive', []) if isinstance(item, dict)}
    return {n for n in names if n and n not in {'_sheet', '_source_row'} | derived}


def _query_many(sheet_names, params, load, sheets):
    """Run one query over several sheets that share the referenced columns.

    Rows gain _sheet and _source_row so results say where each came from.
    """
    names = [n for n, m in sheets.items() if m.get('sheet_type') == 'data'] if sheet_names == ['*'] else sheet_names
    unknown = [n for n in names if n not in sheets]
    if unknown or not names:
        return {'error': f'Unknown sheets {unknown}; use exact names from list_sheets or ["*"] for all data sheets.'}
    needed = _referenced_columns(params)
    frames, queried, skipped = [], [], {}
    for name in names:
        frame = load(name)
        missing = sorted(needed - set(frame.columns))
        if missing:
            skipped[name] = f'missing columns {missing}'
            continue
        source = _source_rows(frame, sheets[name], range(len(frame)))
        source += [None] * (len(frame) - len(source))
        frames.append(Table(['_sheet', '_source_row'] + frame.columns,
                            [{'_sheet': name, '_source_row': n, **row} for n, row in zip(source, frame.rows)],
                            numeric=frame.numeric | {'_source_row'}))
        queried.append(name)
    if not frames:
        common = sorted(set.intersection(*[set(load(n).columns) for n in names])) if names else []
        return {'error': f'No queried sheet has all referenced columns {sorted(needed)}. Columns shared by these '
                         f'sheets: {common}. The tab name is the virtual column "_sheet" '
                         '(e.g. group_by ["_sheet"]).', 'skipped_sheets': skipped}
    combined = concat(frames)
    params = dict(params)
    if params.get('select') and not params.get('aggregations'):
        params['select'] = ['_sheet', '_source_row'] + [c for c in params['select'] if c not in {'_sheet', '_source_row'}]
    result = query_table(combined, params)
    indices = result.pop('contributing_row_indices') if result['aggregated'] else result['result_row_indices']
    result.pop('result_row_indices', None)
    result.pop('contributing_row_indices', None)
    by_sheet = {}
    for i in indices:
        by_sheet.setdefault(combined.rows[i]['_sheet'], []).append(combined.rows[i]['_source_row'])
    result['sources'] = [{'sheet': n, 'source_rows': rows} for n, rows in by_sheet.items()]
    result['sheets_queried'] = queried
    result['skipped_sheets'] = skipped
    return result


def execute_table_tool(
    tool_name: str,
    tool_input: dict,
    s3_client,
    s3_bucket: str,
    s3_prefix: str,
    plan_domain: str,
    filename: str,
    metadata: dict,
) -> dict:
    sheet_name = tool_input.get("sheet_name")
    sheets = metadata.get("sheets", {})

    if tool_name == "query_table" and "sheet_names" in tool_input:
        sheet_names = tool_input["sheet_names"]
        if not isinstance(sheet_names, list) or not all(isinstance(n, str) for n in sheet_names):
            return {"error": 'sheet_names must be a list of sheet names, or ["*"] for all data sheets.'}
        def load(name):
            return load_table(s3_client, s3_bucket, s3_prefix, plan_domain, filename, name, sheet_meta=sheets[name])
        result = _query_many(sheet_names, {k: v for k, v in tool_input.items() if k not in {'sheet_names', 'sheet_name'}},
                             load, sheets)
        return sanitize_result(result)

    if tool_name in {"read_sheet", "query_table"} and sheet_name not in sheets:
        return {"error": "Provide an exact sheet_name from list_sheets."}

    if tool_name == "query_table":
        df = _load_for_query(lambda name: load_table(s3_client, s3_bucket, s3_prefix, plan_domain, filename,
                                                     name, sheet_meta=sheets[name]), sheet_name, sheets[sheet_name])
        result = query_table(df, {k:v for k,v in tool_input.items() if k != 'sheet_name'})
        indices = result.pop('contributing_row_indices') if result['aggregated'] else result['result_row_indices']
        result.pop('result_row_indices', None)
        result.pop('contributing_row_indices', None)
        result['sources'] = [{'sheet': sheet_name,
                              'source_rows': _source_rows(df, sheets[sheet_name], indices),
                              'csv_record_indices': indices,
                              'row_number_basis': 'CSV indices are zero-based; source rows are one-based'}]
        result['warnings'] = sheets[sheet_name].get('warnings', [])
        if result['matched_rows'] == 0 and tool_input.get('filters'):
            needed = _referenced_columns(tool_input)
            others = [n for n, m in sheets.items() if n != sheet_name and m.get('sheet_type') == 'data'
                      and needed <= {c['name'] for c in m.get('columns', [])}]
            if others:
                result['notes'].append(f'No rows matched in {sheet_name!r}, but {len(others)} other sheets have the '
                                       f'same columns: {others}. Query them with "sheet_names": ["*"] before '
                                       'concluding the value is absent.')

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
            indices = sorted({r[key] for r in refs if r[key] is not None})
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
        if smeta.get("sheet_type") == "metadata" and {"source_row", "source_column", "text"} <= set(df.columns):
            # Narrative sheets read as one line per sheet row, with embedded table headers applied.
            lines = [{"source_row": number, "text": text} for number, text in narrative_rows(df.rows)]
            end = min(offset + limit, len(lines))
            return sanitize_result({"sheet_name": sheet_name, "offset": offset, "total_rows": len(lines),
                                    "rows": lines[offset:end], "next_offset": end if end < len(lines) else None,
                                    "warnings": smeta.get("warnings", [])})
        end = min(offset + limit, len(df))
        result = {"sheet_name": sheet_name, "offset": offset, "total_rows": len(df),
                  "rows": df.take(range(offset, end)).records(),
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
                needle = term.casefold()
                searched = [c for c in cols_to_search if c in df.columns]
                matched = [i for i, row in enumerate(df.rows)
                           if any(row[c] is not None and needle in str(row[c]).casefold() for c in searched)]
                if matched:
                    shown = df.take(matched[:50]).select([c for c in col_names if c in df.columns])
                    results[name] = shown.records()
                    source = smeta.get('source_rows', [])
                    details[name] = {'matches': len(matched), 'returned': len(shown),
                                     'truncated': len(matched) > 50,
                                     'source_rows': [source[i] for i in shown.index if i < len(source)]
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
