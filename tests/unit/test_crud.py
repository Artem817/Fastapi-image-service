import pytest

@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("incr_value, filename, content_type, expected_status, expected_msg", [
    (1, "test_1.jpg", "image/jpeg", 200, "uploaded"),
    (2, "test_2.jpg", "image/jpeg", 200, "uploaded"),
    (3, "test_3.txt", "text/plain", 400, "File must be an image"),
    (4, "large.jpg", "image/jpeg", 413, "File too large"),
    (5, "large_1.jpg", "image/jpeg", 413, "File too large"),
    (6, "test_rate.jpg", "image/jpeg", 429, "Too many requests. Maximum 5 images per hour. Please wait 120 seconds."),
    (7, "test_rate_2.jpg", "image/jpeg", 429, "Too many requests. Maximum 5 images per hour. Please wait 120 seconds."),
])
@pytest.mark.anyio
async def test_upload_image_scenarios(
    ac, incr_value, shelf_image, mock_redis_t, mock_redis_b, auth_user, 
    filename, content_type, expected_status, expected_msg
):
    mock_redis_t.incr.return_value = incr_value
    mock_redis_t.ttl.return_value = 120 if incr_value > 5 else 0
    
    if expected_status == 413:
        file_content = b"0" * (10 * 1024 * 1024 + 1)
    elif expected_status == 400:
        file_content = b"not an image content"
    else:
        file_content = shelf_image.getvalue()

    response = await ac.post("/images/upload", files={"file": (filename, file_content, content_type)})

    assert response.status_code == expected_status
    res_data = response.json()
    
    if expected_status == 200:
        assert res_data["status"] == expected_msg
        mock_redis_b.set.assert_called_once()
    else:
        assert res_data["detail"] == expected_msg
        
    if expected_status != 400:
        mock_redis_t.incr.assert_called()
        if incr_value == 1:
            mock_redis_t.expire.assert_called_once()
        if incr_value > 5:
            mock_redis_t.ttl.assert_called_once()
        
