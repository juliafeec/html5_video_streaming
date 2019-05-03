function readURL(input) {
  if (input.files && input.files[0]) {

    var reader = new FileReader();

    reader.onload = function(e) {
      $('.image-upload-wrapPopUp').hide();

      // To be changed by a slide show
      $('.uploaded-image').attr('src', e.target.result);
      $('.image-showIn-wrapPopUp').show();

      //$('.image-title').html(input.files[0].name);
    };

    reader.readAsDataURL(input.files[0]);

  } else {
    removeUpload();
  }
}

function submit() {
  if (document.getElementById('name').value == "") {
  alert("Enter a Name for the Picture");
  } else {
    // when everything is filled, then we can proceed to save the image
    img_src = $('.file-uploaded-image').attr('src');
    var img = document.createElement('img');
    img.src = img_src;

    document.getElementById("image-upload-wrap").appendChild(img);
    closePopUpForm();
  }
}

var windowObjectReference;
var strWindowFeatures = "menubar=yes,location=yes,resizable=yes,scrollbars=yes,status=yes";

function openRequestedPopup() {
  document.getElementById('abc').style.display = "block";
}

function closePopUpForm() {
  document.getElementById('abc').style.display = "none";
}

function openNewUserPopUp(){
  document.getElementById('newUser').style.display = "block";
}

function closeNewUserPopUpForm() {
  document.getElementById('newUser').style.display = "none";
}
