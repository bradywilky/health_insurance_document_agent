import io
import json
from unittest.mock import Mock

import pandas as pd
import pytest

import xlsx_csv_agent as agent
from local_data import prepare_local_file
from llm import LLM


@pytest.fixture
def inputs(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("Region,Revenue\nWest,100\nEast,200\nWest,150\n", encoding="utf-8")
    return prepare_local_file(path)


def test_real_graph_computes_sum(inputs, monkeypatch):
    decisions = iter([
        {"tool": "list_sheets", "parameters": {}},
        {"tool": "get_sheet_schema", "parameters": {"sheet_name": "Sheet1"}},
        {"tool": "analyze_data", "parameters": {"sheet_name": "Sheet1", "query": "Total revenue"}},
        {"tool": "answer", "parameters": {}},
    ])
    monkeypatch.setattr(agent, "_call_llm", lambda *a, **kw: (next(decisions), {}))
    monkeypatch.setattr(agent, "_call_coder_llm", lambda *a, **kw: "result = df['Revenue'].sum()")
    def synthesize(query, findings, **kw):
        assert "450" in findings
        assert "File: sales.csv" in findings
        return "Total revenue is 450.", {}
    monkeypatch.setattr(agent, "_call_synthesis_llm", synthesize)
    assert agent.run_plan_tables_agent(**inputs, query="Total revenue?") == "Total revenue is 450."


def test_stream_executes_once(inputs, monkeypatch):
    decision = Mock(return_value=({"tool": "answer", "parameters": {}}, {}))
    synthesis = Mock(return_value=("Done", {}))
    monkeypatch.setattr(agent, "_call_llm", decision)
    monkeypatch.setattr(agent, "_call_synthesis_llm", synthesis)
    events = list(agent.run_plan_tables_agent_stream(**inputs, query="Summarize"))
    assert events[-1] == {"answer": "Done"}
    assert decision.call_count == synthesis.call_count == 1


def test_step_limit_exceeds_default_graph_limit(inputs, monkeypatch):
    decision = Mock(return_value=({"tool": "list_sheets", "parameters": {}}, {}))
    monkeypatch.setattr(agent, "_call_llm", decision)
    def synthesize(query, findings, **kw):
        assert "Step limit reached" in findings
        return "Incomplete", {}
    monkeypatch.setattr(agent, "_call_synthesis_llm", synthesize)
    assert agent.run_plan_tables_agent(**inputs, query="Keep going") == "Incomplete"
    assert decision.call_count == agent.MAX_STEPS


def test_invalid_json_retries(monkeypatch):
    responses = iter(["not JSON", {"tool": "answer", "parameters": {}}])
    def run(self, **kwargs):
        self.output_raw = "not JSON"
        return next(responses), {"inputTokens": 2}
    monkeypatch.setattr(LLM, "run", run)
    result, usage = agent._call_llm("system", [])
    assert result["tool"] == "answer"
    assert usage["inputTokens"] == 4


def test_search_literal_and_numeric(tmp_path):
    path = tmp_path / "values.csv"
    path.write_text("Name,Code\nA.B,123\nAxB,456\n", encoding="utf-8")
    inputs = prepare_local_file(path)
    metadata = agent.node_init({**inputs, "query": "test"})["metadata"]
    result = agent._execute_tool("search_all_sheets", {"term": "A.B"}, **inputs, metadata=metadata)
    assert len(result["Sheet1"]) == 1
    result = agent._execute_tool("search_all_sheets", {"term": "456"}, **inputs, metadata=metadata)
    assert result["Sheet1"][0]["Name"] == "AxB"


def test_excel_sheets(tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Value": [10, 20]}).to_excel(writer, sheet_name="First", index=False)
        pd.DataFrame({"Value": [30]}).to_excel(writer, sheet_name="Second", index=False)
    inputs = prepare_local_file(path)
    state = agent.node_init({**inputs, "query": "test"})
    assert list(state["metadata"]["sheets"]) == ["First", "Second"]
    assert agent._load_df(**inputs, sheet_name="Second")["Value"].sum() == 30


def test_s3_key_unchanged():
    s3 = Mock()
    s3.get_object.return_value = {"Body": io.BytesIO(b"Value\n42\n")}
    assert agent._load_df(s3, "bucket", "prefix/", "domain", "book.xlsx", "Sales").iloc[0, 0] == 42
    s3.get_object.assert_called_once_with(Bucket="bucket", Key="prefix/domain/preprocessed/book.xlsx/Sales.csv")


def test_empty_parallel_synthesis(inputs):
    result = agent._execute_tool("parallel_synthesize", {}, **inputs, metadata={"sheets": {}})
    assert "error" in result


def test_full_llm_wrapper_and_graph(inputs, monkeypatch):
    from llm import LLMCallLog
    responses = iter([
        '{"tool": "list_sheets", "parameters": {}}',
        '{"tool": "get_sheet_schema", "parameters": {"sheet_name": "Sheet1"}}',
        '{"tool": "analyze_data", "parameters": {"sheet_name": "Sheet1", "query": "Total"}}',
        "result = df['Revenue'].sum()",
        '{"tool": "answer", "parameters": {"response": "Computed total"}}',
        "Total revenue in sales.csv, Sheet1 is 450.",
    ])
    def invoke(self, system, messages):
        self.output_raw = next(responses)
        self.usage = {"inputTokens": 10, "outputTokens": 5}
    monkeypatch.setattr(LLM, "_invoke_llm", invoke)
    log = LLMCallLog()
    answer = agent.run_plan_tables_agent(**inputs, query="Total?", call_log=log)
    assert "450" in answer
    assert len(log.records) == 6
    assert "450" in log.records[-1]["messages"][0]["content"][0]["text"]


def test_repeated_invalid_decision_fails(monkeypatch):
    monkeypatch.setattr(LLM, "run", lambda *a, **kw: ("bad JSON", {}))
    with pytest.raises(ValueError, match="invalid tool decision twice"):
        agent._call_llm("system", [])


def test_empty_worksheet(tmp_path):
    path = tmp_path / "empty.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame().to_excel(writer, sheet_name="Empty", index=False)
    inputs = prepare_local_file(path)
    assert agent._load_df(**inputs, sheet_name="Empty").empty


def test_analysis_comprehension_and_nulls(monkeypatch):
    monkeypatch.setattr(agent, "_call_coder_llm", lambda *a, **kw:
                        "result = {c: int(df[c].count()) for c in df.columns}")
    frame = pd.DataFrame({"Label": ["A", None]})
    result = agent._analyze_data("Count non-null values", frame)
    assert result == "{'Label': 1}"
    assert frame["Label"].isna().sum() == 1
