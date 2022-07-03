from django.shortcuts import render
from django.views.generic import TemplateView

class MirrorBase(TemplateView):
    template_name = 'base.html'
