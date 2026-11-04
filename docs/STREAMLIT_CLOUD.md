# Streamlit Community Cloud

Deploy `AdvayV/Emotion_Detection`, branch `main`, with `dashboard/app.py` as the
main file. Select **Python 3.12** under **Advanced settings** when creating the
app so that deployment matches the verified runtime.

## Dependency installation

The repository-root `requirements.txt` installs `-e .[dashboard]`. This installs
the project and selects its dashboard extra from `pyproject.toml`, including
Streamlit, pandas, Plotly, Excel support and the Python Graphviz package. Plain
installation of the project selects only its core dependencies; it does not
install the dashboard extra. This caused the hosted `import plotly.express`
failure.

Community Cloud searches the entrypoint directory before the repository root.
Do not add a competing dependency file under `dashboard/`: it would take
precedence over the root requirements file. The project metadata remains in
`pyproject.toml`; `requirements.txt` is the Cloud installation entrypoint.

After pushing dependency changes to the deployed branch, Community Cloud
normally reinstalls dependencies automatically. Check **Manage app → Logs** for
installation failures. If the app still shows the previous error, reboot it
from **Manage app** and verify that its repository, branch and main file match
the values above.

## Available without model downloads

The default deployment includes the dashboard and its rule-based demonstrations.
It does **not** install PyTorch/Transformers or download trained checkpoints.
The repository ignores `models/`, so checkpoints installed on a developer's PC
are not included in a GitHub checkout. Missing models are displayed as
unavailable; model comparisons and trained inference require compatible model
files on the host.

For trained inference, provision the exported checkpoints and tokenizer/metadata
files separately, and explicitly install the project's `model` extra in an
environment with sufficient memory and storage. Merely adding the model extra
does not supply weights. Optional Qwen review also requires a reachable Ollama
service; a developer's local Ollama service is not available inside Community
Cloud. Training remains a separate Colab/Kaggle workflow.

## Reproduce the dashboard environment locally

From the repository root, using Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip check
.\.venv\Scripts\python -m streamlit run dashboard/app.py
```

On macOS/Linux, use `.venv/bin/python` in place of
`.\.venv\Scripts\python`. Open the Emotion lab, Analyze text, Dataset explorer
and Overview workspaces. The Overview diagrams use `st.graphviz_chart`, which
requires the Python `graphviz` package even when passed DOT text. These diagrams
are rendered by Streamlit in the browser and do not require a server-side
Graphviz executable or `packages.txt`.

## References

- [Community Cloud dependency selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Community Cloud deployment settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit Graphviz chart requirements](https://docs.streamlit.io/develop/api-reference/charts/st.graphviz_chart)
