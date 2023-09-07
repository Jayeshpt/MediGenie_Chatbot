from django.urls import path
from.import views
from django.views.decorators.csrf import ensure_csrf_cookie

urlpatterns = [
    path('',views.index,name='index'),
    path('upload/', views.upload_file, name='upload_file'),
    path('data_table/', views.show_data, name='show_data'),
    path('chatbot/',views.chatbot,name='chatbot'),
    path('get_response', views.get_response, name='get_response'),
    path('how_to_use/',views.how_to_use,name='how_to_use'),
    path('prompts/',views.prompts,name='prompts'),
    path('cleanup/', ensure_csrf_cookie(views.cleanup), name='cleanup'),


]