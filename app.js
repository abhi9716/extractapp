
    $("#send").click(function(){
        var text = $(this).val();
        $.ajax({
            url: "/process",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({"text": text}),
            success: function(response) {
        $("#res").html(response);
      },
        }).done(function(data) {
            console.log(data);
        });
    });
