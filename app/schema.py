"""
schema.py
=========
Pydantic models for request validation and response formatting.
"""

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """
    Schema for customer input data used for cluster prediction.

    Attributes:
        Age: Customer's age (e.g., 30)
        Income: Annual household income (e.g., 50000)
        Total_Spending: Total amount spent on products (e.g., 2000)
        Children: Number of children in the household (e.g., 1)
        Education: Education level encoded as integer
                   (0=Basic, 1=Diploma, 2=Graduation, 3=Master, 4=PhD)
    """
    Age: int = Field(..., ge=10, le=120, description="Customer's age")
    Income: float = Field(..., ge=0, description="Annual household income")
    Total_Spending: float = Field(..., ge=0, description="Total spending on products")
    Children: int = Field(..., ge=0, le=10, description="Number of children")
    Education: int = Field(
        ..., ge=0, le=4,
        description="Education level: 0=Basic, 1=Diploma, 2=Graduation, 3=Master, 4=PhD"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Age": 30,
                    "Income": 50000,
                    "Total_Spending": 2000,
                    "Children": 1,
                    "Education": 2,
                }
            ]
        }
    }


class PredictionOutput(BaseModel):
    """
    Schema for the prediction response.

    Attributes:
        cluster: The predicted cluster number (0, 1, or 2)
        category: Human-readable category label
    """
    cluster: int = Field(..., description="Predicted cluster number")
    category: str = Field(..., description="Customer category label")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "cluster": 1,
                    "category": "Medium Value Customer",
                }
            ]
        }
    }
