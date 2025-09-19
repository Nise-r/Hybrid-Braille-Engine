from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.Model import get_braille

class BrailleInput(BaseModel):
    text: str

class BrailleOutput(BaseModel):
    text: str

app = FastAPI()


@app.post("/api/v1/braille", response_model=BrailleOutput)
async def braille_endpoint(item: BrailleInput):
    try:
        corrected_text = ""
        for text in item.text.split(" "):    
            corrected_text += get_braille(text)+" "

        return BrailleOutput(text=corrected_text)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred in the AI module."
        )
