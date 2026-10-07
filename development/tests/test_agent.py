import io
from unittest.mock import Mock
import pandas as pd
from backend.storage.local import prepare_local_file
from backend.storage.tables import load_table
from backend.tools.tables import execute_table_tool
from backend.preprocessing.documents import document_from_table_inputs


def test_search_literal_and_numeric(tmp_path):
    path = tmp_path / "values.csv"
    path.write_text("Name,Code\nA.B,123\nAxB,456\n", encoding="utf-8")
    inputs = prepare_local_file(path)
    metadata = document_from_table_inputs(**inputs).table_metadata
    result = execute_table_tool("search_all_sheets", {"term": "A.B"}, **inputs, metadata=metadata)
    assert len(result["Sheet1"]) == 1
    result = execute_table_tool("search_all_sheets", {"term": "456"}, **inputs, metadata=metadata)
    assert result["Sheet1"][0]["Name"] == "AxB"


def test_excel_sheets(tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Value": [10, 20]}).to_excel(writer, sheet_name="First", index=False)
        pd.DataFrame({"Value": [30]}).to_excel(writer, sheet_name="Second", index=False)
    inputs = prepare_local_file(path)
    state = {"metadata": document_from_table_inputs(**inputs).table_metadata}
    assert list(state["metadata"]["sheets"]) == ["First", "Second"]
    assert load_table(**inputs, sheet_name="Second").column("Value") == [30]


def test_s3_key_unchanged():
    s3 = Mock()
    s3.get_object.return_value = {"Body": io.BytesIO(b"Value\n42\n")}
    assert load_table(s3, "bucket", "prefix/", "domain", "book.xlsx", "Sales").rows == [{"Value": 42}]
    s3.get_object.assert_called_once_with(Bucket="bucket", Key="prefix/domain/preprocessed/book.xlsx/Sales.csv")


def test_empty_worksheet(tmp_path):
    path = tmp_path / "empty.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame().to_excel(writer, sheet_name="Empty", index=False)
    inputs = prepare_local_file(path)
    assert len(load_table(**inputs, sheet_name="Empty")) == 0
