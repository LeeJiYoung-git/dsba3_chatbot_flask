# app.py
from flask import Flask, request, render_template, send_from_directory
import torch
import joblib
from gtts import gTTS
import os

app = Flask(__name__)
app.debug=True

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER
  

kogpt2_tokenizer = joblib.load('./kogpt2_tokenizer.pkl')
kogpt2_pretrained = joblib.load('./kogpt2_pretrained.pkl')
kogpt2_finetuning = torch.load('./kogpt2_finetuning.pt', map_location=torch.device('cpu'))


# 학습 전 kogpt2 모델 테스트
def before_gpt_test(text, tokenizer, model):
    input_ids = tokenizer.encode(text)  # 입력 문장을 토크나이즈하여 숫자 ID로 변환
    gen_ids = model.generate(torch.tensor([input_ids]),   # kogpt모델로 문장 생성
                            max_length=128,   # 생성할 문장의 최대 길이
                            repetition_penalty=2.0,  # 반복 방지 가중치
                            pad_token_id=tokenizer.pad_token_id,  # 패딩 토큰 ID
                            eos_token_id=tokenizer.eos_token_id,   # 종료 토큰 ID
                            bos_token_id=tokenizer.bos_token_id,   # 시작 토큰 ID
                            use_cache=True) #캐시 사용 여부
    generated = tokenizer.decode(gen_ids[0,:].tolist())  #생성된 숫자 ID들을 다시 문장으로 디코딩
    return generated  # 생성된 문장 출력

# 학습 후 kagpt2 모델 테스트
def after_gpt_test(q, tokenizer, model):
    Q_TKN = "<usr>"  # 사용자 질문 토큰
    A_TKN = "<sys>"
    SENT = '<unused1>'
    EOS = '</s>'
    with torch.no_grad():
        q = q.strip()
        a = ""
        while 1:
            input_ids = torch.LongTensor(tokenizer.encode(Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0)
            pred = model(input_ids)   # 모델에 입력을 주어 대답 생성
            pred = pred.logits.cpu()
            gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]  #모델 출력에서 다음 토큰을 선택해 대답 생성함
            if gen == EOS:  # 대답 토큰이 종료 토큰과 일치될 때(대답 생성이 종료될 때)
                break 
            a += gen.replace("▁", " ")
        text = a.strip()
        tts = gTTS(text=text, lang='ko')   # 생성된 대답을 음성으로 변환
        tts.save("./static/sound/answer.wav")  
    return text

@app.route('/get_audio')
def get_audio():
    return send_from_directory('./static/sound', 'answer.wav')

# Main page
@app.route('/')
def start():
    return render_template('main.html')


# 콜센터 페이지
@app.route('/callcenter', methods=['GET', 'POST'])
def callCenter():
    text = ''
    before_answer = ''
    after_answer = ''
    if request.method == 'POST':
        text = request.form.get('message')  # 입력 문장
        before_answer = before_gpt_test(text, kogpt2_tokenizer, kogpt2_pretrained)
        after_answer = after_gpt_test(text, kogpt2_tokenizer, kogpt2_finetuning)
    return render_template('callcenter.html', text=text, 
                           before_answer=before_answer, after_answer=after_answer,
                           audio_file_url='/get_audio')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)