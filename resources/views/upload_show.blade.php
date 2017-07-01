<html>
<body>
<h1>上传图片</h1>
<form action="http://120.77.84.254:8810/about"  enctype="multipart/form-data" method="post" >
    <input type="hidden" name="_token" value="{{ csrf_token() }}" />
    <input type="file" name="photo" />
    <input type="submit" value="提交" />
</form>
</body>
</html>
