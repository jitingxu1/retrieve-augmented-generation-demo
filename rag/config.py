from pydantic import BaseModel

class AppConfig(BaseModel):
    app_id: str
    user_id: str