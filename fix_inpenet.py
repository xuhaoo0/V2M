'''
用VolumetricSMPL检测并修复是否穿模（或者PoseShield）

args:
- smpl_file，例如a/b/c.pt

输出：a/c.json，字段"interpenetration"
- "exist, N"表示存在穿模的错误，并且判定在第N帧穿模
- "nonexist"表示不存在

输出：a/b/c_fix.pt，表示修复后的smpl参数
'''