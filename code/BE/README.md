## Structure

```
├─BE
│  dense.py
│  logger.py
│  main.py
│  pyproject.toml
│  __init__.py
```

## How to use
[한국어 링크](https://mycogno.notion.site/BE-5b57a6ac24264fa6b5f43f5bd6e0eed0)

You have to set up poetry and environment variable GOOGLE_APPLICATION_CREDENTIALS

* You have to get key file! (Our key file is only openned to my team). key file's location is  ```/opt/ml/final-project-level3-nlp-11/code/BE/key/```
* You have to install poetry! You can install poetry with ```curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -``` command
---
1. Approach to bash.bashrc with ```vim /etc/bash.bashrc``` command
2. Paste the command to bottom of file.
```
source $HOME/.poetry/env
export GOOGLE_APPLICATION_CREDENTIALS="/opt/ml/final-project-level3-nlp-11/code/BE/key/
```
3. Restart the terminal
4. in ```/code/BE```, install packages with ```poetry install``` command
