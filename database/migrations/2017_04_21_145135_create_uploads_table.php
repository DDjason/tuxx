<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateUploadsTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        //
        Schema::create('uploads',function(Blueprint $table){
            $table->engine = 'InnoDB';
            $table->increments('id');
            $table->string('old_name');
            $table->string('new_name');
            $table->string('phy_address');
            $table->timestamp('created_at');
            $table->timestamp('updated_at');
            $table->string('ip');
            $table->integer('train_ok')->default(0);
            $table->string('requset');
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        //
        Schema::drop('uploads');
    }
}
