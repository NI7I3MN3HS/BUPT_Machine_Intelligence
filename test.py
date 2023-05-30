import openai

openai.api_key = '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'

class InteractiveStory:
    def __init__(self):
        self.story_type = None
        self.story_background = None
        self.main_characters = None
        self.story_content = ''
        self.choices = []
        self.states = {}

    def set_story_type(self, story_type: str):
        self.story_type = story_type

    def set_story_background(self, story_background: str):
        self.story_background = story_background

    def set_main_characters(self, main_characters: list):
        self.main_characters = main_characters

    def generate_characters_description(self):
        for character in self.main_characters:
            prompt = f"为{character}生成一个形象描述和性格特点："
            response = openai.Completion.create(
              engine="text-davinci-003",
              prompt=prompt,
              temperature=0.5,
              max_tokens=500
            )
            character_description = response.choices[0].text.strip()
            self.story_content += f"{character}是这样的人：{character_description}\n"

    def get_story(self):
        return self.story_content
    
    def user_interaction(self, prompt: str, choices: list):
        print(prompt)
        for i, choice in enumerate(choices, start=1):
            print(f"{i}. {choice}")

        user_choice = int(input("你的选择是："))
        self.choices.append(choices[user_choice - 1])


    def generate_choices(self, prompt):
        # 生成选项
        options_prompt = prompt + "现在，有哪些可能的选项？"
        options_response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=options_prompt,
            temperature=0.5,
            max_tokens=500
        )

        # 将生成的选项划分为列表
        choices = options_response.choices[0].text.strip().split('\n')

        # 校验选项
        valid_choices = [choice for choice in choices if self.validate_choice(choice)]

        # 如果所有的选项都被过滤掉了，那就重新生成选项
        if not valid_choices:
            valid_choices = self.generate_choices(prompt)

        return valid_choices

    def validate_choice(self, choice):
        # 这是一个简单的校验逻辑：选项必须至少包含5个字符
        # 你可以根据你的具体需求来修改这个逻辑
        return len(choice) >= 5

    def generate_story(self):
        prompt = f"这是一个{self.story_type}的故事，背景设定在{self.story_background}，主要角色有{', '.join(self.main_characters)}。"
        prompt += self.story_content
        prompt += "然后故事发生了什么？"

        while True:
            # 生成故事片段
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.5,
                max_tokens=500
            )

            self.story_content += response.choices[0].text.strip()
            print(self.story_content)

            # 生成选项
            valid_choices = self.generate_choices(prompt)

            self.user_interaction("你会如何应对？", valid_choices)

            prompt = f"你选择了{self.choices[-1]}，然后呢？"

story = InteractiveStory()
story.set_story_type("科幻")
story.set_story_background("未来的地球")
story.set_main_characters(["沈白", "胡天"])
story.generate_characters_description()
story.generate_story()
print(story.get_story())