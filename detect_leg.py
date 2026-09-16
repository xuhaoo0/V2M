'''
检测左右腿是否互换

args:
- smpl_file，例如a/b/c.pt

输出：a/c.json，字段"leg"
- “true, N"表示存在左右腿互换的错误，并且判定在第N帧开始互换
- "false"表示不存在
'''