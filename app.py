from fastapi import FastAPI

app = FastAPI()


@app.get('/')
def root():
    return 'aaa'


@app.get('/isim/{isim}')
def selamver(isim: str):
    return f'selammmm {isim}'


