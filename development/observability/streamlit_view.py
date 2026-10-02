"""Streamlit panel for one question's captured model calls (see capture.py)."""
import json

import streamlit as st


def show_llm_calls(summary, key):
    """Totals, a call-by-call timeline, and the full prompt and response of a chosen call."""
    calls, totals = summary['calls'], summary['totals']
    if not calls:
        return
    with st.expander(f"Model calls ({totals['llm_calls']} · {totals['total_tokens']:,} tokens)"):
        columns = st.columns(5)
        columns[0].metric('Model calls', totals['llm_calls'])
        columns[1].metric('Input tokens', f"{totals['input_tokens']:,}")
        columns[2].metric('Output tokens', f"{totals['output_tokens']:,}")
        columns[3].metric('Largest call', f"#{totals['largest_call']}",
                          f"{calls[totals['largest_call'] - 1]['total_tokens']:,} tokens", delta_color='off')
        columns[4].metric('Time in model', f"{totals['model_ms'] / 1000:.1f} s",
                          f"of {totals['duration_ms'] / 1000:.1f} s total", delta_color='off')
        st.dataframe([{k: c[k] for k in ('call', 'step', 'start_ms', 'gap_ms', 'duration_ms', 'input_tokens',
                                          'output_tokens', 'total_tokens', 'stop_reason')}
                      | {'tool_calls': ', '.join(c['tool_calls'])} for c in calls], hide_index=True)
        if summary['tools']:
            st.caption('Tool executions')
            st.dataframe(summary['tools'], hide_index=True)
        number = st.selectbox('Inspect call', [c['call'] for c in calls], key=f'{key}-call',
                              format_func=lambda n: f"#{n} {calls[n - 1]['step']} "
                                                    f"({calls[n - 1]['total_tokens']:,} tokens)")
        call = calls[number - 1]
        if not call['input']:
            st.caption('Prompt text was not captured (TRACE_CONTENT is off).')
        for label, messages in (('Prompt', call['input']), ('Response', call['output'])):
            if messages:
                st.markdown(f'**{label}**')
                for message in messages:
                    _show_message(message)


def _show_message(message):
    st.caption(message.get('role', ''))
    if message.get('content'):
        st.code(message['content'], language=None, wrap_lines=True)
    for call in message.get('tool_calls') or []:
        st.code(f"{call['name']}({json.dumps(call.get('arguments'), indent=2)})", language='json',
                wrap_lines=True)
    for result in message.get('tool_results') or []:
        st.code(result.get('result', ''), language='json', wrap_lines=True)
