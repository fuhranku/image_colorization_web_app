$(function(){

});

$('#ajax').on('submit',function(e){
    e.preventDefault(); // Disables submit's default action
    const imgData = new FormData($('#ajax').get(0));
    console.log(imgData);
    $.ajax({
        type: "POST",
        url: colorize_url,
        data: imgData,
        processData: false,
        contentType: false,
        success: function (response) {
            console.log(response);
            $('.image-container').append(
                `<img src=${response.url}>`
            );
        },
        error: function(response){
            console.log('AJAX ERROR');
            console.log(response);
        }
    });
})

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

