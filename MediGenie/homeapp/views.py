from django.shortcuts import render

# Create your views here.
def index(request):
    return  render(request,'index.html')

### ---------------------------------------- File upload and Accespting --------------------------------------------------

from django.conf import settings
import os
import pandas as pd


df = None  # Define the df variable in the outer scope

def upload_file(request):
    global df  # Use the global keyword to access the outer scope df variable
    
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        
        # Check if the file extension is allowed
        allowed_extensions = ['.csv', '.xlsx', '.xls']
        if not any(uploaded_file.name.endswith(ext) for ext in allowed_extensions):
            return render(request, 'upload.html', {'error_message': 'Invalid file format'})

        # Save the file to the media directory
        media_root = settings.MEDIA_ROOT
        file_path = os.path.join(media_root, uploaded_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        # Read the uploaded file into the df variable
        try:
            df = pd.read_csv(file_path)  # Modify this based on your actual file format
        except Exception as e:
            return render(request, 'upload.html', {'error_message': f'Error reading file: {e}'})
        dataset_size = len(df)
        
        # Pass the dataset size to the template context
        context = {
            'dataset_size': dataset_size,
        }
        print('jkjlkj',context)
        return render(request, 'index.html', context) 
    
    return render(request, 'upload.html')

###----------------------------------------------------- End of File upload and Accespting --------------------------------------------------
