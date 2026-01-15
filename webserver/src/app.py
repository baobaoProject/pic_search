import logging
import os
import os.path as path
import shutil
import multiprocessing

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from flask_restful import reqparse
from werkzeug.utils import secure_filename

from common.config import DATA_PATH, DEFAULT_TABLE
from common.config import UPLOAD_PATH
from service.count import do_count
from service.delete import do_delete
from service.search import do_search, do_text_search
from service.train import do_train, train_status_cache
from indexer.index import milvus_client
import gunicorn.app.base

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['jpg', 'png','jpeg',"gif","bmp"])
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['JSON_SORT_KEYS'] = False
CORS(app)


@app.route('/api/v1/train', methods=['POST'])
def do_train_api():
    args = reqparse.RequestParser(). \
        add_argument('Table', type=str, location=['args', 'form', 'json']). \
        add_argument('File', type=str, location=['args', 'form', 'json']). \
        add_argument('VectorDimension', type=str, location=['args', 'form', 'json']). \
        add_argument('EmbeddingIndexType', type=str, location=['args', 'form', 'json']). \
        parse_args()
    table_name = args['Table']
    file_path = args['File']
    embedding_index_type = args['EmbeddingIndexType']
    try:
        # 在 Eager 模式下，不需要传递 graph 和 sess
        result = do_train(table_name, file_path, embedding_index_type)
        return result
    except Exception as e:
        return "Error with {}".format(e)


@app.route('/api/v1/delete', methods=['POST'])
def do_delete_api():
    args = reqparse.RequestParser(). \
        add_argument('Table', type=str, location=['args', 'form', 'json']). \
        parse_args()
    table_name = args['Table']
    print("delete table.")
    status = do_delete(table_name)
    try:
        shutil.rmtree(DATA_PATH)
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
    except:
        print("cannot remove", DATA_PATH)
    return "{}".format(status)


@app.route('/api/v1/count', methods=['POST'])
def do_count_api():
    args = reqparse.RequestParser(). \
        add_argument('Table', type=str, location=['args', 'form', 'json']). \
        parse_args()
    table_name = args['Table']
    rows = do_count(table_name)
    return "{}".format(rows)


@app.route('/api/v1/process')
def train_status_api():
    cache = train_status_cache()
    return "current: {}, total: {}".format(cache.get('current', 0), cache.get('total', 0))


@app.route('/api/v1/data/<path:image_name>', methods=['GET'])
def image_path(image_name):
    file_name = str("/" + image_name)
    print("image_name:" + file_name)
    if path.exists(file_name):
        return send_file(file_name)
    return "file not exist", 404


@app.route('/api/v1/search', methods=['POST'])
def do_search_api():
    args = reqparse.RequestParser(). \
        add_argument("Table", type=str, location=['args', 'form']). \
        add_argument("Num", type=int, default=1, location=['args', 'form']). \
        parse_args()

    table_name = args['Table']
    if not table_name:
        table_name = DEFAULT_TABLE
    top_k = args['Num']
    file = request.files.get('file', "")
    if not file:
        return "no file data", 400
    if not file.name:
        return "need file name", 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        res_id,res_distance = do_search(table_name, file_path, top_k)
        if isinstance(res_id, str):
            return res_id
        res_img = [request.url_root +"api/v1/data" + x for x in res_id]
        res = dict(zip(res_img,res_distance))
        res = sorted(res.items(),key=lambda item:item[1])
        return jsonify(res), 200
    return "not found", 400


@app.route('/api/v1/text_search', methods=['POST'])
def do_text_search_api():
    """Text-to-image search API"""
    args = reqparse.RequestParser(). \
        add_argument("Table", type=str, location=['args', 'form', 'json']). \
        add_argument("Text", type=str, location=['args', 'form', 'json']). \
        add_argument("Num", type=int, default=1, location=['args', 'form', 'json']). \
        parse_args()
    
    table_name = args['Table']
    if not table_name:
        table_name = DEFAULT_TABLE
    text = args['Text']
    if not text:
        return "no text data", 400
    top_k = args['Num']
    
    try:
        res_id, res_distance = do_text_search(table_name, text, top_k)
        if isinstance(res_id, str):
            return res_id
        res_img = [request.url_root + "api/v1/data" + x for x in res_id]
        res = dict(zip(res_img, res_distance))
        res = sorted(res.items(), key=lambda item: item[1])
        return jsonify(res), 200
    except Exception as e:
        logging.error(f"Error in text search: {e}")
        return str(e), 500


@app.before_request
def before_request():
    # 定义允许的前缀列表
    passed_prefixes = ['/api/v1/data']

    # 检查当前请求路径是否符合允许的前缀
    request_path = request.path
    # 使用logging，将请求地址、请求客户端ip，请求时间、请求方法拼起来打印

    logging.info(request.remote_addr+","+request.method+","+ request.url)
    # 如果当前请求路径在指定集合中，则跳过连接客户端
    is_allowed = any(request_path.startswith(prefix) for prefix in passed_prefixes)
    if  is_allowed:
        return
    # 在每个请求之前建立Milvus连接
    # Establish connection to Milvus
    milvus_client()

class StandaloneApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, app, options=None):
        self.application = app
        self.options = options or {}
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                  if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

if __name__ == "__main__":
    # 设置多进程的启动方式为spawn，解决CUDA在子进程中的初始化问题
    multiprocessing.set_start_method('spawn', force=True)
    
    # 配置日志级别为 INFO，确保 logging.info 能够输出到控制台
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    # app.run(host="0.0.0.0", debug=False)
    # 使用生产级服务器
    options = {
        'bind': '0.0.0.0:5000',
        'workers': 1,      # 单进程，确保全局变量共享和显存节省
        'threads': 4,      # 可选：开启多线程以提高 I/O 并发能力（处理搜索请求时有用）
        'timeout': 120,
    }
    StandaloneApplication(app, options).run()