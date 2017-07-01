<?php

namespace App;

use Illuminate\Database\Eloquent\Model;

class Upload extends Model
{
    //
    //使用的表
    protected $table = 'uploads';

    //不使用Laravel的默认时间
    public $timestamps = true;

    protected $fillable = array('old_name','new_name','phy_address','ip');



}
