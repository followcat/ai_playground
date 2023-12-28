# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install fastapi uvicorn
"""
import sys
import os
import asyncio
import uvicorn
import argparse
from functools import partial

from fastapi import FastAPI, Query
from starlette.middleware.cors import CORSMiddleware
from loguru import logger

sys.path.append('..')
from nerpy import NERModel

from pick_name import pick_NER

pwd_path = os.path.abspath(os.path.dirname(__file__))
# Use fine-tuned model
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="shibing624/bert4ner-base-chinese",
                    help="Model save dir or model name")
args = parser.parse_args()
s_model = NERModel('bert', args.model_name_or_path)

# define the app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


@app.get('/')
async def index():
    return {"message": "index, docs url: /docs"}


@app.get('/pickner')
async def pickner(q: str = Query(..., min_length=1, max_length=99999999, title='query')):
    try:
        #preds, outputs, entities = s_model.predict([q], split_on_space=False)
        #result_dict = {'entity': entities}
        loop = asyncio.get_running_loop()
        result_dict = await loop.run_in_executor(
            None,
            partial(pick_NER, q)
        )
        logger.debug(f"Successfully get sentence entity, q:{q}")
        return result_dict
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8001)
