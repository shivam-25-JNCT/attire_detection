from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

   
    APP_NAME:str = "Corporate Attendance & Dress Code Detection Service"
    DEBUG:bool = False

    # model path
    PERSON_MODEL_PATH:str=(r"F:\person_detection\corporate_detection\models\person.pt")
    ID_CARD_MODEL_PATH:str=(r"F:\person_detection\corporate_detection\models\best_id.pt")
    CLIP_MODEL_PATH:str=(r"openai/clip-vit-base-patch32")

    PERSON_CONF_THRESHOLD:float=0.50
    ID_CARD_CONF_THRESHOLD:float=0.50

    DEVICE:str='cuda'
    SAVE_SNAPSHOT:bool=True
    SNAPSHOT_DIR: Path = Path("snapshots")
    DEFAULT_RTSP_URL: str =r"F:\person_detection\corporate_detection\source\clip_main.mp4"


    class CONFIG:
        env_file=".env"
        extra="ignore"


settings=Settings()
