# 🌍 Multilingual Translator  

Этот проект представляет собой web-приложение для автоматического перевода текста между различными языками с использованием модели `mBART` от Facebook AI. Приложение создано на `Streamlit` и позволяет быстро и удобно выполнять переводы без необходимости установки дополнительного программного обеспечения.  

## 🚀 Функции  

- Поддержка 50 языков благодаря модели `mbart-large-50-many-to-many-mmt`. 
- Быстрый и точный перевод текста любой длины.  
- Возможность обратного перевода для проверки результата.  
- Простота использования через web-интерфейс без установки.  

## 🛠️ Установка и запуск  

### 📥 Установка зависимостей  

Сначала клонируйте репозиторий и установите необходимые библиотеки:  

```bash
git clone https://github.com/yourusername/multilingual-translator.git  
cd multilingual-translator  
pip install -r requirements.txt  
▶ Запуск приложения
Для локального запуска выполните:

bash
streamlit run app.py  

🏗️ Архитектура
Архитектура приложения включает несколько компонентов:

Модель mBART – предварительно обученная нейросеть для перевода.
Streamlit – фреймворк для создания веб-интерфейса.
Python + Hugging Face Transformers – библиотека для загрузки и работы с моделью.
Браузер – клиентская часть, через которую пользователь взаимодействует с приложением.

🖥️ Развертывание в облаке
Приложение можно развернуть в облаке, например, на Streamlit Community Cloud, Heroku или Google Cloud Run.

Пример развертывания на Streamlit Sharing:

Создайте репозиторий на GitHub и загрузите код.
Перейдите на Streamlit Cloud и подключите репозиторий.
Настройте переменные среды, если требуется.
Запустите и получите публичную ссылку.
