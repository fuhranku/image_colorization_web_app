$(function(){

});

$('#ajax').on('submit',function(e){
    e.preventDefault(); // Disables submit's default action
    const imgData = new FormData($('#ajax').get(0));
    console.log(imgData)
    $.ajax({
        type: "POST",
        url: colorize_url,
        data: imgData,
        processData: false,
        contentType: false,
        beforeSend: function(){
            $('.spinner').addClass('show');
            $('body').prepend(`
                <div class="overlay"></div>
            `)
        },
        complete: function () {
            $('.spinner').removeClass('show');
            $('body').find('.overlay').remove();
        },
        success: function (response) {  
            console.log(response);
            $('.image-preview').append(
                `<img src=${response.url}>`
            );
        },
        error: function(response){
            console.log('AJAX ERROR');
            console.log(response);
        }
    });
})

$('input[type="file"]').on('change',function(){
    setThumbnail();
});

    // Drag enter
$('.file-uploader').on('dragenter', function (e) {
    e.stopPropagation();
    e.preventDefault();
});

// Drag over
$('.file-uploader').on('dragover', function (e) {
    e.stopPropagation();
    e.preventDefault();
});

// Drop
$('.file-uploader').on('drop', function (e) {
    e.stopPropagation();
    e.preventDefault();
    $('input[type="file"]').prop('files',e.originalEvent.dataTransfer.files);
    setThumbnail();
});

function setThumbnail(){
    // Clean previewer container
    $('.image-preview').empty();
    // Check for valid file. If so, put preview
    var file = $('input[type="file"]').get(0).files[0];
    if(IsValidImage(file)){
        var reader = new FileReader();
        reader.onload = function(){
            $(".image-preview").append(`
                <img src=${reader.result}>
            `);
        }
        reader.readAsDataURL(file);
    }else{
        $(".image-preview").append(`
            <p class="invalid-image"> Sorry, this file is not supported</p>
        `);
    }
}

function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}

function IsValidImage(file){
    // if (!file) return false;
    var fileType = file["type"];
    var validImageTypes = ["image/gif", "image/jpeg", "image/png", "image/jpg",];
    if ($.inArray(fileType, validImageTypes) < 0) return false;
    return true;
}
