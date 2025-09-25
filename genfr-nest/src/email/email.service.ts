import { MailerService } from '@nestjs-modules/mailer';
import { BadRequestException, Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class EmailService {
    constructor(private readonly mailService: MailerService,
        private prisma: PrismaService
    ) {}

    private COOLDOWN_VERIFY = 1;
    private COOLDOWN_PASSWORD = 2;



    async sendVerificationMail(email:string, verifyUrl:string){
        await this.checkCooldown(email, 'verify-email',this.COOLDOWN_VERIFY);
        
        await this.mailService.sendMail({
            to: email,
            subject: 'Verify your email',
            template: './verify-email', // automatically adds .hbs
            context: { name: 'User', verifyUrl, year: new Date().getFullYear() },
            from: '"Generator Frajdy" <noreply@yourdomain.com>',
        })

        await this.updateLastSent(email, 'verify-email');
    }
    async sendPasswordResetMail(email:string, resetUrl:string){
        await this.checkCooldown(email, 'reset-password',this.COOLDOWN_PASSWORD);

        await this.mailService.sendMail({
            to: email,
            subject: 'Reset your password',
            template: './reset-password',
            context: { name:  'User', resetUrl, year: new Date().getFullYear() },
            from: '"Generator Frajdy" <noreply@yourdomain.com>',
        })
        
        await this.updateLastSent(email, 'reset-password');
    }
    private async checkCooldown(userEmail: string, action: string, cooldownMinutes:number) {
    const record = await this.prisma.emailAction.findUnique({
      where: { userEmail_action: { userEmail, action } },
    });

    if (record) {
      const diff = Date.now() - record.lastSentAt.getTime();
      if (diff < cooldownMinutes * 60_000) {
        throw new BadRequestException(
          `You can only request ${action} every ${cooldownMinutes} minutes`
        );
      }
    }
  }

  private async updateLastSent(userEmail: string, action: string) {
    await this.prisma.emailAction.upsert({
      where: { userEmail_action: { userEmail, action } },
      update: { lastSentAt: new Date() },
      create: { userEmail, action, lastSentAt: new Date() },
    });
  }
}
