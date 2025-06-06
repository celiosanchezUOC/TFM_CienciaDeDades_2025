# TFM_CienciaDeDades_2025

## Agents d'Intel·ligència Artificial amb Anàlisi Automatitzada d'Imatges Dermatoscòpiques per ajudar amb el Diagnòstic

**Autor:** Celio Sánchez Bañuls  
**Tutor:** Samuel Paul Gallegos Serrano  
**Universitat:** UOC  
**Any:** 2025

---

### 📄 Descripció

Aquest projecte és el resultat del Treball de Fi de Màster en Ciència de Dades on l'objectiu principal del projecte es el desenvolupament d’una eina automatitzada per a la classificació d’imatges dermatoscòpiques mitjançant models avançats d’intel·ligència artificial. 
L'entorn serà capaç de classificar imatges dermatoscòpiques de lesions cutànies com a benignes o malignes, proporcionant suport tant a professionals mèdics com a usuaris no especialitzats en la detecció precoç del càncer de pell.
El sistema integra diverses tecnologies per facilitar la seva aplicació en entorns clínics i investigadors.

---

### Prerequisits:
Es necessiten diferents ferramentes per tal de poder executar el projecte.
* Visual studio Code - https://code.visualstudio.com/
* Docker - https://www.docker.com/
* n8n - https://n8n.io/
* FastApi - https://fastapi.tiangolo.com/
* Streamlit - https://streamlit.io/
  
Tambe s'han de descarregar els conjunts de dades que utilitzarem.
* HAM10000 - https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
* Stanford Diverse Dermatology Images -  https://ddi-dataset.github.io/index.html
* Fitzpatrick17k - https://github.com/mattgroh/fitzpatrick17k

Hem utilitzat aquestes llibreries:
```
numpy==1.26.0
pandas==2.2.0
opencv-python==4.7.0.72
matplotlib==3.8.0
seaborn==0.13.0
scikit-learn==1.3.0
imbalanced-learn==0.11.0
tensorflow==2.19.0
tensorflow-addons==0.22.0
torch
torchvision
torchaudio
vit-keras==0.1.0
Pillow==10.0.0
datasets==2.14.0
fastapi
uvicorn
python-multipart
timm
streamlit
requests
plotly
python-dotenv
```

## Scripts inicials per als models i conjunt de dades
### Undersampling
Este script hi ha que adaptarlo segons el dataset (en este cas es sobre el dataset de Stanford Diverse Dermatology Images)
- ProcesarDatasetUndersampling.py
### Entrenament
- DenseNet201.py
- InceptionResNetV2.py
- SwinTransformerV2.py
- ViT-B16.py
- Xception.py
---

Hem fet el undersampling de cada Dataset , tenim els conjunts de dades triats per als nostres entrenaments on hem utilitzat cada script per a entrenar cada model en cada classificació.
Una vegada tenim els 15 models entrenats 5 per cada classificació , ja disposem del necessari per tal de treballar amb l'eina de automatització.

---
## 🚀 Estructura del projecte de Docker
```
TFM_celiosanchez/
│
├── .n8n/                      # Configuració i workflows d'automatització n8n
├── __pycache__/               # Fitxers temporals Python
├── app/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── models/                # Carpeta per als models IA (vegeu nota important)
│ 		├── Classif_1_ SwinV2.pth
│ 		├── Classif_1_DenseNet201.h5
│ 		├── Classif_1_inceptionresnetv2.h5
│ 		├── Classif_1_ViT-B16.pth
│ 		├── Classif_1_Xception.h5
│ 		├── Classif_2_DenseNet201.h5
│ 		├── Classif_2_inceptionresnetv2.h5
│ 		├── Classif_2_SwinV2.pth
│ 		├── Classif_2_ViT-B16.pth
│ 		├── Classif_2_Xception.h5
│ 		├── Classif_3_DenseNet201.h5
│ 		├── Classif_3_inceptionresnetv2.h5
│ 		├── Classif_3_SwinV2.pth
│ 		├── Classif_3_ViT-B16.pth
│ 		└── Classif_3_Xception.h5
│
├── main.py                    # Backend FastAPI
├── Dockerfile                 # Imatge Docker principal
├── docker-compose.yml         # Orquestració de serveis
├── requirements.txt           # Dependències Python
├── Prediure_Imatge.json       # Workflow n8n (importable)
├── streamlit_app.py           # Interfície d'usuari Streamlit
```

---

## 🧩 Components principals

- **Docker & docker-compose:** Faciliten el desplegament de tota la infraestructura amb una sola comanda.
- **n8n:** Automatitza el flux de treball de recepció i processament d’imatges.
- **FastAPI:** Servei backend per a la inferència dels models IA.
- **Streamlit:** Interfície web per a l’usuari final.
- **Models IA:** Diverses arquitectures preentrenades (SwinV2, DenseNet201, InceptionResNetV2, ViT-B16, Xception).

---

## ⚠️ Nota important sobre els models IA

**Per motius d'espai, la carpeta `app/models/` dins del ZIP no conté els fitxers reals dels models entrenats.**  
Els models són de gran mida i estan allotjats externament a Google Drive.

- Pots sol·licitar l'accés als models reals per fer proves a través d’aquest enllaç:  
  👉 [Models IA a Google Drive](https://drive.google.com/drive/folders/1TuQ7kD3UrR3Wz_BppZfWx-SjZmHnq3Y0?usp=sharing)
- Un cop obtinguts, només cal col·locar-los a la carpeta `app/models/` per poder utilitzar tota la funcionalitat del projecte.

---

## 🛠️ Instal·lació i execució

1. **Clona el repositori**
- git clone https://github.com/celiosanchezUOC/TFM_CienciaDeDades_2025.git
- cd TFM_CienciaDeDades_2025

2. **Descomprimeix `TFM_celiosanchez.zip`**
- unzip TFM_celiosanchez.zip
- cd TFM_celiosanchez

3. **Descarrega els models IA des del Google Drive**  
- Sol·licita accés i descarrega els fitxers. Col·loca’ls a `app/models/`.

4. **Desplega els serveis amb Docker Compose**
- docker-compose up --build

5. **Accedeix als serveis:**
- **Streamlit:** [http://localhost:8501](http://localhost:8501)
- **FastAPI docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **n8n:** [http://localhost:5678](http://localhost:5678)

6. **Importa el workflow a n8n**
- Accedeix a n8n i importa el fitxer `Prediure_Imatge.json` per tenir el flux d’automatització preparat.

---

## 🖥️ Com fer una predicció

1. Obre la interfície Streamlit.
2. Carrega una imatge dermatoscòpica.
3. Selecciona el model desitjat.
4. Visualitza la predicció i la probabilitat associada.

---

## 🧑‍💻 Estructura de carpetes

- `.n8n/`: Configuració i historial de workflows d’automatització.
- `app/models/`: Carpeta per als models d’IA (vegeu nota important).
- `main.py`: Backend amb FastAPI.
- `streamlit_app.py`: Interfície d’usuari.
- `Prediure_Imatge.json`: Workflow d’automatització per n8n.
- `Dockerfile` i `docker-compose.yml`: Per desplegar tot el sistema.
- `requirements.txt`: Llista de dependències Python.

---

## 📑 Documentació addicional

- Consulta la memòria del TFM per a detalls tècnics, resultats i justificació de decisions.
- Per qualsevol dubte, contacta amb l’autor via [GitHub Issues](https://github.com/celiosanchezUOC/TFM_CienciaDeDades_2025/issues).

---

## 📬 Contacte

**Celio Sánchez Bañuls**  
[GitHub](https://github.com/celiosanchezUOC)  

---

*Projecte desenvolupat per al Màster Universitari en Ciència de Dades (UOC, 2025).*
