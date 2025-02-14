from locust import HttpUser, task

class StressUser(HttpUser):
    host = "http://localhost:8000"  # Add this line with your API's host URL
    
    @task
    def predict_argentinas(self):
        self.client.post(
            "/predict", 
            json={
                "flights": [
                    {
                        "OPERA": "Aerolineas Argentinas", 
                        "TIPOVUELO": "N", 
                        "MES": 3
                    }
                ]
            }
        )

    @task
    def predict_latam(self):
        self.client.post(
            "/predict", 
            json={
                "flights": [
                    {
                        "OPERA": "Grupo LATAM", 
                        "TIPOVUELO": "N", 
                        "MES": 3
                    }
                ]
            }
        )