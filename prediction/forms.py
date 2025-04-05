from django import forms
from .models import MRIImage

class MRIUploadForm(forms.ModelForm):
    class Meta:
        model = MRIImage
        fields = ['file']
