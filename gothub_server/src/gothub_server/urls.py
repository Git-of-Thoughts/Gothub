"""
URL configuration for gothub_server project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from webhook_process.views import github_payload, home, take_order_web

urlpatterns = [
    path("", home, name="home"),
    path("admin/", admin.site.urls),
    path("payload/", github_payload, name="webhook"),
    # path("oauth/callback/", register_user, name="user_register"),
    path("take_order_web/", take_order_web, name="take_order_web"),
]
