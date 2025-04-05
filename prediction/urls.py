from django.urls import path
from .views import home, upload_mri, predict, results

urlpatterns = [
    path('', home, name='home'),
    path('upload/', upload_mri, name='upload_mri'),
    path('predict/', predict, name='predict'),
    path('results/', results, name='results'),
]
