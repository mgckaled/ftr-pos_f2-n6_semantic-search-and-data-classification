<!--markdownlint-disable-->

# Instruções de Instalação

## Dependências Faltantes no Pipfile

As seguintes dependências precisam ser adicionadas ao Pipfile na **raiz do repositório**:

```bash
cd ../../  # Voltar para raiz do repositório

# Instalar dependências faltantes
pipenv install pillow tqdm kagglehub
```

### Explicação das Dependências

| Pacote | Uso no Projeto |
|--------|----------------|
| `pillow` | Carregamento e manipulação de imagens (PIL/Image) |
| `tqdm` | Progress bars para loops longos |
| `kagglehub` | Download automático do dataset do Kaggle |

## Verificação

Após instalar, verifique se tudo está ok:

```bash
pipenv shell
python -c "from PIL import Image; from tqdm import tqdm; import kagglehub; print('✅ Todas as dependências instaladas!')"
```

## Pipfile Completo Sugerido

Se preferir, adicione manualmente ao `[packages]` do Pipfile:

```toml
[packages]
# ... (dependências existentes)
pillow = "*"
tqdm = "*"
kagglehub = "*"
```

Depois execute:

```bash
pipenv install
```

## Notas

- O projeto foi testado com Python 3.11.9
- Todas as outras dependências já estão no Pipfile original
- Se encontrar problemas com `torch`, pode ser necessário instalar a versão CPU:
  ```bash
  pipenv install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
