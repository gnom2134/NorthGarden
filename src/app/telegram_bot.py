from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from src.pipeline.common_pipeline import Pipeline
from dotenv import load_dotenv
from pathlib import Path
import logging
import os


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
pipeline = None
logger = logging.getLogger(__name__)


def start(update, context):
    update.message.reply_text("Type your query.")


def help(update, context):
    update.message.reply_text("South park dialog generation bot. Try it yourself.")


def reply(update, context):
    text = update.message.text
    response = ""
    if len(text) > 0 and pipeline is not None:
        response = pipeline.process_query(text)
    else:
        response = "Got empty input."

    update.message.reply_text(response)
    logger.log(response, update)


def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    global pipeline

    load_dotenv(Path("./environments/app.env"))
    updater = Updater(os.getenv("BOT_TOKEN"), use_context=True)
    pipeline = Pipeline(
        os.getenv("CHARS", "Cartman,Kyle,Stan").split(","),
        gen_type=os.getenv("GEN_TYPE", "stump"),
        cl_type=os.getenv("CL_TYPE", "stump"),
        iterations=int(os.getenv("ITERATIONS", 2)),
        cl_model_name=os.getenv("CL_MODEL_NAME", None),
        gen_model_names=os.getenv("GEN_MODEL_NAMES", None).split(",")
        if os.getenv("GEN_MODEL_NAMES", None) is not None
        else None,
        cl_model_path=os.getenv("CL_MODEL_PATH", None),
        gen_model_paths=os.getenv("GEN_MODEL_PATHS", None).split(",")
        if os.getenv("GEN_MODEL_PATHS", None) is not None
        else None,
    )

    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    dp.add_handler(MessageHandler(Filters.text, reply))

    dp.add_error_handler(error)

    updater.start_polling()

    updater.idle()


if __name__ == "__main__":
    main()
