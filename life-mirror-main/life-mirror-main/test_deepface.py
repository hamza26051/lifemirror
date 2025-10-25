import os
os.environ['LIFEMIRROR_MODE'] = 'prod'
os.environ['FACE_USE_DEEPFACE'] = 'true'

from src.tools.face_tool import FaceTool, ToolInput

tool = FaceTool()
result = tool.run(ToolInput(media_id='test', url='test_person.jpg'))

print('Tool result:', result.success)
print('Faces found:', len(result.data.get('faces', [])))
for i, face in enumerate(result.data.get('faces', [])):
    print(f'Face {i} attributes:', face.get('attributes', {}))
    print(f'Face {i} bbox:', face.get('bbox'))
    print(f'Face {i} crop_url:', face.get('crop_url'))