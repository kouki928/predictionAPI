from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# モデルのロード
Amodel = pickle.load(open("./models/Amodel.pkl", "rb"))
Lmodel = pickle.load(open("./models/Lmodel.pkl", "rb"))
Pmodel = pickle.load(open("./models/Pmodel.pkl", "rb"))

class UserInfo(BaseModel):
    APreviousDayCompletion: float
    APreviousDayTarget: float
    AWeeklyCompletion: float
    Age: str
    Frequency: str
    Gender: str
    Goal: str
    Height: str
    LPreviousDayCompletion: float
    LPreviousDayTarget: float
    LWeeklyCompletion: float
    PPreviousDayCompletion: float
    PPreviousDayTarget: float
    PWeeklyCompletion: float
    Weight: str

def parsonSelection(data):
    # 目標BMIに基づく理想体重の算出
    goal_bmi = {
        "Male": {"MuscleStrength": 25.0, "WeightLoss": 22.0, "HealthMaintenance": 24.0},
        "Female": {"MuscleStrength": 23.0, "WeightLoss": 20.0, "HealthMaintenance": 22.0}
    }

    # 辞書から必要な情報を取得
    Gender = data["Gender"]
    Frequency = data["Frequency"]
    Age = int(data["Age"])
    Goal = data["Goal"]
    Height = int(data["Height"])
    Weight = float(data["Weight"])

    # 目標BMIから理想体重を計算
    ideal_bmi = goal_bmi[Gender][Goal]
    idealWeight = round(ideal_bmi * (Height / 100) ** 2, 1)

    # 文字列を数値に置換
    gender_mapping = {"Male": 1, "Female": 2, "Other": 3}
    goal_mapping = {"MuscleStrength": 1, "WeightLoss": 2, "HealthMaintenance": 3}
    frequency_mapping = {"Low": 1, "Moderate": 2, "High": 3}
    Gender, Goal, Frequency = map(lambda x, mapping: mapping.get(x, x), [Gender, Goal, Frequency], [gender_mapping, goal_mapping, frequency_mapping])

    # 部位ごとの目標回数を格納する辞書
    TargetRepsDict = {}

    # A
    APreviousDayCompletion = float(data["APreviousDayCompletion"])
    AWeeklyCompletion = float(data["AWeeklyCompletion"])
    APreviousDayTarget = float(data["APreviousDayTarget"])
    # L
    LPreviousDayCompletion = float(data["LPreviousDayCompletion"])
    LWeeklyCompletion = float(data["LWeeklyCompletion"])
    LPreviousDayTarget = float(data["LPreviousDayTarget"])
    # P
    PPreviousDayCompletion = float(data["PPreviousDayCompletion"])
    PWeeklyCompletion = float(data["PWeeklyCompletion"])
    PPreviousDayTarget = float(data["PPreviousDayTarget"])

    # 新しいデータの構築
    new_data = np.array([[Gender, Frequency, Age, Goal, Height, Weight, idealWeight, APreviousDayCompletion, AWeeklyCompletion, APreviousDayTarget, LPreviousDayCompletion, LWeeklyCompletion, LPreviousDayTarget, PPreviousDayCompletion, PWeeklyCompletion, PPreviousDayTarget]], dtype=np.float32)

    # 予測＆格納
    keys = ["A", "L", "P"]
    TargetRepsDict = {}
    models = {"A": Amodel, "L": Lmodel, "P": Pmodel}
    category = {"A": "AbsTraining", "L": "LegTraining", "P": "PectoralTraining"}
    for model in models:
        # 推論
        prediction = models[model].predict(new_data)[0]
        # 推論結果を格納
        TargetRepsDict[category[model]] = round(prediction, 1)

    return TargetRepsDict

@app.post("/target/")
def bmi_prediction(req: UserInfo):
    data = req.dict()
    
    # 関数を呼び出して結果を取得
    result = parsonSelection(data)
    
    return result

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="127.0.0.1", port=8000)