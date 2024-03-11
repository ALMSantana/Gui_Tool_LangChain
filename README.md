## 🔑 Gerar API_KEY e associar ao .env

### criar arquivo .env e associar a chave da api

```python
OPENAI_API_KEY = "SUA_CHAVE_AQUI"
```

## ⚙️ Configuração do Ambiente

### venv no Windows:

```bash
python -m venv curso_lang
curso_lang\Scripts\activate
```

### venv no Mac/Linux:

```bash
python3 -m venv curso_lang
source curso_lang/bin/activate
```

## 📚 Ver compatibilidades e requirements.txt

- FAISS compatível com python <= 3.11

```bash
pip install -r requirements.txt
```