import requests
import json

def send_post_request(url, require, text_value):
    # 设置要发送的数据
    data = {
        "type": "together",
        "require": require,
        "text": text_value
    }

    # 发送 POST 请求
    response = requests.post(url, json=data)

    # 检查响应状态码并输出结果
    if response.status_code == 200:
        print("Success:")
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    else:
        print(f"Failed with status code: {response.status_code}")
        print(response.text)

if __name__ == '__main__':
    # 定义服务器地址和端口
    url = 'http://10.136.198.30:9992/fix_table'  # 修改为你的服务器端口

    # 要发送的 text 数据
    require_value = "把第一列和第二列合并为一列"
    text_value = "<tr><th>项</th><th>指</th><th>标</th><th>检验方法</th></tr><tr><td>高锰酸钾(KMnO₄)含量，w/%</td><td>99.0~100.5</td><td>附录 A 中 A.4</td><td></td></tr><tr><td>氧化物(以 Cl 计)，w/%</td><td>≤</td><td>0.01</td><td>附录 A 中 A.5</td></tr>"

    # 发送请求并打印结果
    send_post_request(url, require_value, text_value)
