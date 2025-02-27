from flask import Flask
import random

app = Flask(__name__)

def randomPassword(seed):
    seed = 100000000 + int(seed)
    # 字母类型
    englishChar = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'l', 'k', 'j', 'h', 'g', 'f', 'd', 's', 'a', 'z',
                   'x',
                   'c', 'v',
                   'b', 'n', 'm']
    # 数字类型
    numberChar = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

    # 符号类型
    symbolChar = ['!', '@', '#', '$']
    passwordCharSet = englishChar.copy() + numberChar.copy() + symbolChar.copy()
    random.seed(seed)
    # 把密码打乱
    random.shuffle(passwordCharSet)

    return "".join(passwordCharSet[:16])

@app.route('/')
@app.route('/<int:number>')
def getPassword(number=1):
    return f"""<h1 align="center">{randomPassword(number)}</h1>"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

