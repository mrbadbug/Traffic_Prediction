from flask import Flask, render_template
import pandas as pd
import threading
import app as rtp

app = Flask(__name__)

@app.route("/")
def home():
    df = pd.read_csv("data/Metro_Interstate_Traffic_Volume.csv").tail(50)
    return render_template("dashboard.html", tables=[df.to_html(classes='data')])

if __name__ == "__main__":
    threading.Thread(target=rtp.realtime_loop).start()
    app.run(debug=True, host="0.0.0.0", port=5000)
