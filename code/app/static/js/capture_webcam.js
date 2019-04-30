'use strict';

var photo = document.getElementById('photo');
var photoContext = photo.getContext('2d');
var startBtn = document.getElementById('startwebcam');
// var canvas = document.getElementById("myCanvas");
// var ctx = canvas.getContext('2d');

const mediaStreamConstraints = {
    video: true,
};

//video element where the stream will be placed
var localVideo = document.querySelector('video');
let localStream;

startBtn.addEventListener('click', startWebcam);

//handles success by adding the MediaStream to the video element
function gotLocalMediaStream(mediaStream){
    localStream = mediaStream;
    localVideo.srcObject = mediaStream;
}

//handles error by logging a message to the console with error message
function handleLocalMediaStreamError(mediaStream){
    console.log('navigator.getUserMedia error: ', error);
}

function startWebcam(){
    navigator.mediaDevices.getUserMedia(mediaStreamConstraints)
        .then(gotLocalMediaStream).catch(handleLocalMediaStreamError);
}

function snapPhoto(){
    // Draws current image from the video element into the canvas
    // ctx.drawImage(localVideo, 0,0, canvas.width, canvas.height);
    // var dataURL = canvas.toDataURL('image/jpeg', 1.0);
    $.post("/image", {
        image: dataURL
    });
}

function capturePhotos(){
    setInterval(snapPhoto, 1000);
}

startBtn.addEventListener('click', capturePhotos);
// startBtn.onclick = function(){
//     setInterval(snapPhoto, 1000);
// }

function overlayImage(){
    var height = $("#camera").height();
    var width = $("#camera").width();
    var top = $("#camera").offset().top;
    var left = $("#camera").offset().left;
    $('#image').css(
        "height", height);
    $('#image').css("top", top);
    $('#image').css("left", left);
    $('#image').css(
        "width", width);
    $('#image').show();  
    $.get("/svg", function(data, status){
    //alert("Data: " + data + "\nStatus: " + status);
    $("#image").attr("src", data);
    setTimeout(overlayImage, 500);
    //overlayImage();
    });

}
overlayImage();