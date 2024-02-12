import cv2
image_path = 'cars.jpg'

start_point = (150,420)
end_point = (1100,420)


image = cv2.imread(image_path)
image = cv2.line(image, start_point, end_point, (0,155,0), 3) 
cv2.imshow('cars',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
