from flask import Flask, Response
import yolo_app

app = Flask(__name__)

def generate():
    # 動画ファイルを開く
    with open('output.avi', 'rb') as video_file:
        while True:
            # 動画のバイナリデータを読み込んでストリームに書き込む
            data = video_file.read(1024)
            if not data:
                break
            yield data

@app.route('/navi-ken-video')
def index():
    # 動画ストリームをレスポンスとして返す
    return Response(generate(), mimetype='video/x-msvideo')

if __name__ == '__main__':
    app.run(debug=True)
