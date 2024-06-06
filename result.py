#Image Generation
#Generate an Image from a Text Description

def generate_image(text_description):
    sequence, _ = custom_tokenizer([text_description])
    padded_sequence = pad_sequences_custom(sequence, max_length)

    noise = np.random.normal(0, 1, (1, 100))

    generated_image = generator.predict([noise, padded_sequence])
    generated_image = (generated_image + 1) / 2.0

    return generated_image[0]

#Example Usage

text_description = input("Enter a text description: ")
generated_image = generate_image(text_description)

plt.imshow(generated_image)
plt.axis('off')
plt.show()
