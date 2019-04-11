function readURL(input) {
  if (input.files && input.files[0]) {

    var reader = new FileReader();

    reader.onload = function(e) {
      $('.image-upload-wrapPopUp').hide();

      $('.file-uploaded-image').attr('src', e.target.result);
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

// Everything necessasry to connect to S3
//updatePolicy: function(){
//  var key = this.get('folder') + this.get('filename');
//  this.set({key: key});
//
//  POLICY_JSON = { "expiration": "2012-12-01T12:00:00.000Z",
//          "conditions": [
//          ["eq", "$bucket", this.get('bucket')],
//          ["starts-with", "$key", this.get('key')],
//          {"acl": this.get('acl')},
//          {"success_action_redirect": this.get('successActionRedirect')},
//          {"x-amz-meta-filename": this.get('filename')},
//          ["starts-with", "$Content-Type", this.get('contentType')]
//          ]
//        };
//
//  var secret = this.get('AWSSecretKeyId');
//  var policyBase64 = Base64.encode(JSON.stringify(POLICY_JSON));
//  var signature = b64_hmac_sha1(secret, policyBase64);
//
//  this.set({POLICY: policyBase64 });
//  this.set({SIGNATURE: signature });
//}