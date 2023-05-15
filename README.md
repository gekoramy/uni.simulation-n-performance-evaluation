# uni.simulation-n-performance-evaluation

init python venv

```shell
python -m venv venv
venv/bin/pip install -r requirements.txt
```

populate `.git/config`

```shell
git config --local filter.strip-notebook-output.clean "jq '.cells |= map(if .cell_type == \"markdown\" then (.metadata = {} | del(.outputs?, .execution_count?)) else . end)' | jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```