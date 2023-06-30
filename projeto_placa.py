import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
def encontrarPlaca(source):
    img = cv2.imread(source)
    #cv2.imshow("img",img)

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("cinza",cinza)

    _,bin = cv2.threshold(cinza, 90, 255, cv2.THRESH_BINARY)
    #cv2.imshow("bin",bin)

    desfoque = cv2.GaussianBlur(bin,(5,5),0)
    cv2.imshow("des",desfoque)

    contornos, hierarquia = cv2.findContours(desfoque, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contornos, -1, (0, 255, 0), 1)

    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        if perimetro > 140:
            aprox = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            if len(aprox) == 4:
                (x, y, alt, lar) = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + alt, y + lar), (0, 255, 0), 2)
                placa = img[y:y + lar, x:x + alt]
                cv2.imwrite('placa.png', placa)

    cv2.imshow("contornos", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preProcessamentoPlaca():
    img_placa = cv2.imread("placa.png")

    if img_placa is None:
        return

    img_cinza = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY)
    _,img_binary = cv2.threshold(img_cinza, 90, 255, cv2.THRESH_BINARY)

    cv2.imshow("res", img_binary)
    cv2.imwrite('placaFinal.png', img_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ocrImagemPlaca():
    image = cv2.imread("placaFinal.png")

    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'

    saida = pytesseract.image_to_string(image, lang='eng', config=config)

    return saida

if __name__ == "__main__":
    source = "carro2.png"
    encontrarPlaca(source)

    pre = preProcessamentoPlaca()

    ocr = ocrImagemPlaca()

    print(ocr)