from django.urls import path

from .views import MirrorBase

app_name = "business"
urlpatterns = [
    path('', MirrorBase.as_view(), name='mirror_base'),
]