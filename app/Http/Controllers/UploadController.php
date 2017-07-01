<?php

namespace App\Http\Controllers;

use App\Upload;
use Illuminate\Http\Request;

use App\Http\Requests;
use App\Http\Controllers\Controller;

class UploadController extends Controller
{
    /**
     * Display a listing of the resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function index()
    {
        //
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
        //
    }

    /**
     * Store a newly created resource in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @return \Illuminate\Http\Response
     */

    //jason create_at 20170421
    //接收 http 文件传输
    public function store(Request $request)
    {
        $ip = $request->getClientIp();
        $file = $request->file('photo');
        if($file->isValid()){
            $originalName = $file->getClientOriginalName(); // 文件原名
            $ext = $file->getClientOriginalExtension();     // 扩展名
            $realPath = $file->getRealPath();   //临时文件的绝对路径
            $type = $file->getClientMimeType();     // image/jpeg
            // 上传文件
            $filename = date('H-i-s') . '-' . uniqid() . '.' . $ext;
            //echo $filename;
            // 使用我们新建的uploads本地存储空间（目录）

            $filepath = 'uploads/'.date('Y-m-d');
            $bool = $file -> move($filepath,$filename);;
            $data['old_name'] = $originalName;
            $data['new_name'] = $filename;
            $data['phy_address'] = $filepath.'/'.$filename;
            $data['ip'] = $ip;
            $uploadsuse =  Upload::create($data);

            if($uploadsuse){
                $ret_json['judje'] = 1;
                $ret_json['add_id'] = $uploadsuse->id;

            }else{
                $ret_json['judje'] = 0;
            }
            echo json_encode($ret_json);
        }else{
            $ret_json['judje'] = 0;
            echo json_encode($ret_json);
        }
    }

    /**
     * Display the specified resource.
     *
     * @param  int  $id
     * @return \Illuminate\Http\Response
     */
    public function show($id)
    {
        //
        $uploaduse = Upload::find($id);

        $ret_json['judje'] = 0;//返回结果
        $ret_json['request'] = 'no#answer';
       // var_dump($uploaduse);

        if(!$uploaduse){
            echo json_encode($ret_json);
            exit;
        }

        $ret_answer = $uploaduse->requset;//处理结果



        if($ret_answer){
            $ret_json['judje'] = 1;
//            $ret_json['judjett'] = 'ttt';
            $ret_json['request'] = $ret_answer;
        }else{
            $ret_json['judje'] = 0;

            $phy_address = public_path($uploaduse->phy_address);
	    $exec_query = 'python '.public_path('cifar10/cifar10_eval_00.py').' '.$phy_address.' 2>error.txt';  
	    $last_line = exec($exec_query,$retval);
//	dump($retval);

           //     $ret_json['needcode'] = $exec_query;
            if(!empty($last_line)) {
                $uploaduse->requset = $last_line;
                $uploaduse->save();
                $ret_json['judje'] = 1;
                $ret_json['request'] = $last_line;
            }
        }
      //  $ret_json['request'] = $uploaduse->phy_address;
        echo json_encode($ret_json);

    }

    /**
     * Show the form for editing the specified resource.
     *
     * @param  int  $id
     * @return \Illuminate\Http\Response
     */
    public function edit($id)
    {
        //
    }



    /**
     * Update the specified resource in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @param  int  $id
     * @return \Illuminate\Http\Response
     */
    public function update(Request $request, $id)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     *
     * @param  int  $id
     * @return \Illuminate\Http\Response
     */
    public function destroy($id)
    {
        //
    }
}
