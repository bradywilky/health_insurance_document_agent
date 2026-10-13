import io
from unittest.mock import Mock
import pandas as pd
from backend.storage.tables import TableFolder, load_table
from backend.tools.tables import execute_table_tool
from backend.preprocessing.documents import ingest_document


def test_search_literal_and_numeric():
    doc = ingest_document("values.csv", b"Name,Code\nA.B,123\nAxB,456\n")
    result = execute_table_tool("search_all_sheets", {"term": "A.B"}, doc.table_folder, metadata=doc.table_metadata)
    assert len(result["Sheet1"]) == 1
    result = execute_table_tool("search_all_sheets", {"term": "456"}, doc.table_folder, metadata=doc.table_metadata)
    assert result["Sheet1"][0]["Name"] == "AxB"


def test_excel_sheets(tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Value": [10, 20]}).to_excel(writer, sheet_name="First", index=False)
        pd.DataFrame({"Value": [30]}).to_excel(writer, sheet_name="Second", index=False)
    doc = ingest_document(path.name, path.read_bytes())
    assert list(doc.table_metadata["sheets"]) == ["First", "Second"]
    assert load_table(doc.table_folder, doc.table_metadata["sheets"]["Second"]).column("Value") == [30]


def test_saved_table_reads_its_csv_from_the_document_folder():
    s3 = Mock()
    s3.get_object.return_value = {"Body": io.BytesIO(b"Value\n42\n")}
    folder = TableFolder(s3, "bucket", "base/group/documents/preprocessed/book.xlsx/")
    assert load_table(folder, {"csv_file": "Sales.csv"}).rows == [{"Value": 42}]
    s3.get_object.assert_called_once_with(Bucket="bucket", Key="base/group/documents/preprocessed/book.xlsx/Sales.csv")


def test_empty_worksheet(tmp_path):
    path = tmp_path / "empty.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame().to_excel(writer, sheet_name="Empty", index=False)
    doc = ingest_document(path.name, path.read_bytes())
    assert len(load_table(doc.table_folder, doc.table_metadata["sheets"]["Empty"])) == 0
