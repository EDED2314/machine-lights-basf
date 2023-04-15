from trycourier import Courier
import os
from dotenv import load_dotenv
load_dotenv()


def sendemail():
  auth = os.getenv("AUTH")

  client = Courier(auth_token=f"{auth}")

  resp = client.send_message(
    message={
      "to": {
        "email": "eddietang2314@gmail.com",
      },
      "template": "3VR2W6JPVG4M0DGB8RJYNK57PVK5",
      "data": {
          "message":"testing"
      },
    }
  )

  print(resp['requestId'])