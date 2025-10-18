@echo off
echo === Downloading pre-trained weights ===
curl -L -o models\BERT-Yuri-CLS.zip ^
  https://github.com/ArtistC220/BERT-LLM-Yuri-cls/releases/download/v1.0.0/BERT-Yuri-CLS.zip
powershell -command "Expand-Archive -Force models\BERT-Yuri-CLS.zip models\"
del models\BERT-Yuri-CLS.zip
echo Done!
